from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _LEROBOT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on environment
    LeRobotDataset = None
    recompute_stats = None
    _LEROBOT_IMPORT_ERROR = exc


DEFAULT_ANALYZE_INDICES = [6, 7, 8]
DEFAULT_INACTIVE_VALUE = 0.0
DEFAULT_ATOL = 1e-6


@dataclass
class ActionChunk:
    path: Path
    actions: np.ndarray
    episode_index: np.ndarray
    table: pa.Table | None = None


def _round_float(value: float, decimals: int = 6) -> float:
    return float(np.round(float(value), decimals))


def _require_lerobot() -> tuple[Any, Any]:
    if LeRobotDataset is None or recompute_stats is None:
        raise ImportError(
            "lerobot is required for dataset stats recomputation. "
            "Please initialize the IB_Robot environment before running lerobot_action_gap_repair."
        ) from _LEROBOT_IMPORT_ERROR
    return LeRobotDataset, recompute_stats


def _find_parquet_files(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {dataset_root}")
    parquet_files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return parquet_files


def _load_action_table(
    parquet_path: Path, *, action_column: str, episode_index_column: str
) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_path)
    if table.schema.get_field_index(action_column) < 0:
        raise KeyError(f"Column '{action_column}' not found in {parquet_path}")
    if table.schema.get_field_index(episode_index_column) < 0:
        raise KeyError(f"Column '{episode_index_column}' not found in {parquet_path}")

    actions = np.asarray(table[action_column].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got shape {actions.shape} in {parquet_path}")

    episode_index = np.asarray(table[episode_index_column].to_pylist(), dtype=np.int64)
    if episode_index.ndim != 1 or episode_index.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Expected episode_index to be a 1D array with length {actions.shape[0]}, got shape {episode_index.shape}"
        )

    return table, actions, episode_index


def _count_distribution(values: np.ndarray) -> dict[float, int]:
    rounded_values = np.round(np.asarray(values, dtype=np.float32), 6)
    unique_values, counts = np.unique(rounded_values, return_counts=True)
    return {_round_float(value): int(count) for value, count in zip(unique_values, counts, strict=True)}


def _load_dataset_chunks(
    dataset_root: Path,
    *,
    action_column: str = "action",
    episode_index_column: str = "episode_index",
    include_tables: bool = False,
) -> list[ActionChunk]:
    chunks: list[ActionChunk] = []
    for parquet_path in _find_parquet_files(Path(dataset_root)):
        table, actions, episode_index = _load_action_table(
            parquet_path,
            action_column=action_column,
            episode_index_column=episode_index_column,
        )
        chunks.append(
            ActionChunk(
                path=parquet_path,
                actions=actions,
                episode_index=episode_index,
                table=table if include_tables else None,
            )
        )
    return chunks


def _concatenate_chunks(chunks: list[ActionChunk]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([chunk.actions for chunk in chunks], axis=0),
        np.concatenate([chunk.episode_index for chunk in chunks], axis=0),
    )


def _compute_gap_lengths(
    values: np.ndarray,
    *,
    target_value: float,
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> list[int]:
    values = np.asarray(values, dtype=np.float32)
    is_target = np.isclose(values, target_value, atol=atol)
    changes = np.diff(np.r_[False, is_target, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    gap_lengths: list[int] = []
    for previous_end, next_start in zip(ends[:-1], starts[1:], strict=False):
        gap = next_start - previous_end - 1
        if gap <= 0:
            continue
        gap_values = values[previous_end + 1 : next_start]
        if np.all(np.isclose(gap_values, inactive_value, atol=atol)):
            gap_lengths.append(int(gap))
    return gap_lengths


def analyze_action_array(
    actions: np.ndarray,
    episode_index: np.ndarray,
    *,
    analyze_indices: list[int],
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> dict[int, dict[str, dict[float, dict[int, int]] | dict[float, int]]]:
    actions = np.asarray(actions, dtype=np.float32)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got shape {actions.shape}")
    if episode_index.ndim != 1 or episode_index.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Expected episode_index to be a 1D array with length {actions.shape[0]}, got shape {episode_index.shape}"
        )

    analysis: dict[int, dict[str, dict[float, dict[int, int]] | dict[float, int]]] = {}
    unique_episode_indices = np.unique(episode_index)

    for action_index in analyze_indices:
        if action_index < 0 or action_index >= actions.shape[1]:
            raise IndexError(f"action index {action_index} is out of range for action dim {actions.shape[1]}")

        index_values = actions[:, action_index]
        value_counts = _count_distribution(index_values)
        gap_counts: dict[float, dict[int, int]] = {}
        for target_value in value_counts:
            if np.isclose(target_value, inactive_value, atol=atol):
                continue
            gap_counter: Counter[int] = Counter()
            for current_episode_index in unique_episode_indices:
                episode_values = index_values[episode_index == current_episode_index]
                gap_counter.update(
                    _compute_gap_lengths(
                        episode_values,
                        target_value=target_value,
                        inactive_value=inactive_value,
                        atol=atol,
                    )
                )
            gap_counts[target_value] = dict(sorted(gap_counter.items()))

        analysis[action_index] = {
            "value_counts": dict(sorted(value_counts.items())),
            "gap_counts": dict(sorted(gap_counts.items())),
        }

    return analysis


def analyze_action_parquet_file(
    parquet_path: Path,
    *,
    analyze_indices: list[int],
    action_column: str = "action",
    episode_index_column: str = "episode_index",
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> dict[int, dict[str, dict[float, dict[int, int]] | dict[float, int]]]:
    _, actions, episode_index = _load_action_table(
        parquet_path,
        action_column=action_column,
        episode_index_column=episode_index_column,
    )
    return analyze_action_array(
        actions,
        episode_index,
        analyze_indices=analyze_indices,
        inactive_value=inactive_value,
        atol=atol,
    )


def analyze_dataset(
    dataset_root: Path,
    *,
    analyze_indices: list[int],
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> dict[int, dict[str, dict[float, dict[int, int]] | dict[float, int]]]:
    actions, episode_index = _concatenate_chunks(_load_dataset_chunks(Path(dataset_root)))
    return analyze_action_array(
        actions,
        episode_index,
        analyze_indices=analyze_indices,
        inactive_value=inactive_value,
        atol=atol,
    )


def format_analysis_summary(analysis: dict[int, dict[str, dict[float, dict[int, int]] | dict[float, int]]]) -> str:
    lines: list[str] = []
    for action_index, stats in sorted(analysis.items()):
        lines.append(f"action[{action_index}] value counts:")
        for value, count in stats["value_counts"].items():
            lines.append(f"  {value}: {count}")
        lines.append(f"action[{action_index}] gap counts:")
        if stats["gap_counts"]:
            for target_value, gap_counts in stats["gap_counts"].items():
                if gap_counts:
                    gap_summary = ", ".join(f"{gap}:{count}" for gap, count in gap_counts.items())
                else:
                    gap_summary = "none"
                lines.append(f"  target {target_value}: {gap_summary}")
        else:
            lines.append("  none")
    return "\n".join(lines)


def bridge_target_value_gaps(
    values: np.ndarray,
    gap_threshold: int,
    *,
    target_value: float,
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    if gap_threshold < 0:
        raise ValueError(f"gap_threshold must be >= 0, got {gap_threshold}")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D control array, got shape {values.shape}")

    bridged_values = values.copy()
    is_target = np.isclose(values, target_value, atol=atol)
    changes = np.diff(np.r_[False, is_target, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    for previous_end, next_start in zip(ends[:-1], starts[1:], strict=False):
        gap = next_start - previous_end - 1
        if gap <= 0 or gap > gap_threshold:
            continue
        gap_values = values[previous_end + 1 : next_start]
        if np.all(np.isclose(gap_values, inactive_value, atol=atol)):
            bridged_values[previous_end + 1 : next_start] = target_value

    return bridged_values.astype(np.float32)


def _resolve_processing_targets(
    *,
    indices: list[int] | None,
    target_values: list[float] | None,
) -> tuple[list[int], list[float]]:
    if indices is None or target_values is None:
        raise ValueError("--process-indices and --target-values must both be provided in rewrite mode.")
    if len(indices) != len(target_values):
        raise ValueError("--process-indices and --target-values must have the same length.")
    if not indices:
        raise ValueError("--process-indices must not be empty.")
    return list(indices), [float(value) for value in target_values]


def _validate_rewrite_request(
    chunks: list[ActionChunk],
    *,
    gap_threshold: int,
    indices: list[int] | None,
    target_values: list[float] | None,
    skip_recompute_stats: bool,
) -> tuple[list[int], list[float]]:
    process_indices, process_target_values = _resolve_processing_targets(
        indices=indices,
        target_values=target_values,
    )
    if gap_threshold < 0:
        raise ValueError(f"gap_threshold must be >= 0, got {gap_threshold}")

    action_dim = chunks[0].actions.shape[1]
    for action_index in process_indices:
        if action_index < 0 or action_index >= action_dim:
            raise IndexError(f"action index {action_index} is out of range for action dim {action_dim}")

    if not skip_recompute_stats:
        _require_lerobot()

    return process_indices, process_target_values


def _bridge_dataset_chunks(
    chunks: list[ActionChunk],
    *,
    gap_threshold: int,
    indices: list[int],
    target_values: list[float],
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> None:
    all_actions, all_episode_index = _concatenate_chunks(chunks)
    updated_actions = all_actions.copy()
    for current_episode_index in np.unique(all_episode_index):
        episode_mask = all_episode_index == current_episode_index
        for action_index, target_value in zip(indices, target_values, strict=True):
            updated_actions[episode_mask, action_index] = bridge_target_value_gaps(
                updated_actions[episode_mask, action_index],
                gap_threshold,
                target_value=target_value,
                inactive_value=inactive_value,
                atol=atol,
            )

    start = 0
    for chunk in chunks:
        end = start + chunk.actions.shape[0]
        chunk.actions = updated_actions[start:end]
        start = end


def rewrite_action_parquet_file(
    parquet_path: Path,
    *,
    gap_threshold: int,
    indices: list[int] | None,
    target_values: list[float] | None,
    action_column: str = "action",
    episode_index_column: str = "episode_index",
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    atol: float = DEFAULT_ATOL,
) -> None:
    table, actions, episode_index = _load_action_table(
        parquet_path,
        action_column=action_column,
        episode_index_column=episode_index_column,
    )
    process_indices, process_target_values = _resolve_processing_targets(
        indices=indices,
        target_values=target_values,
    )

    for action_index in process_indices:
        if action_index < 0 or action_index >= actions.shape[1]:
            raise IndexError(f"action index {action_index} is out of range for action dim {actions.shape[1]}")

    updated_actions = actions.copy()
    for current_episode_index in np.unique(episode_index):
        episode_mask = episode_index == current_episode_index
        for action_index, target_value in zip(process_indices, process_target_values, strict=True):
            updated_actions[episode_mask, action_index] = bridge_target_value_gaps(
                updated_actions[episode_mask, action_index],
                gap_threshold,
                target_value=target_value,
                inactive_value=inactive_value,
                atol=atol,
            )

    action_field_index = table.schema.get_field_index(action_column)
    updated_action_array = pa.array(updated_actions.tolist(), type=table.schema.field(action_column).type)
    updated_table = table.set_column(action_field_index, table.schema.field(action_column), updated_action_array)
    pq.write_table(updated_table, parquet_path, compression="snappy")


def _write_chunk_actions(chunks: list[ActionChunk], *, action_column: str = "action") -> None:
    for chunk in chunks:
        if chunk.table is None:
            raise ValueError(f"Missing table metadata for chunk {chunk.path}")
        action_field_index = chunk.table.schema.get_field_index(action_column)
        updated_action_array = pa.array(chunk.actions.tolist(), type=chunk.table.schema.field(action_column).type)
        updated_table = chunk.table.set_column(
            action_field_index,
            chunk.table.schema.field(action_column),
            updated_action_array,
        )
        pq.write_table(updated_table, chunk.path, compression="snappy")


def process_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    gap_threshold: int,
    indices: list[int] | None,
    target_values: list[float] | None,
    repo_id: str | None = None,
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
    skip_recompute_stats: bool = False,
) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    src_chunks = _load_dataset_chunks(src_root)
    process_indices, process_target_values = _validate_rewrite_request(
        src_chunks,
        gap_threshold=gap_threshold,
        indices=indices,
        target_values=target_values,
        skip_recompute_stats=skip_recompute_stats,
    )
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    shutil.copytree(src_root, dst_root)

    dst_chunks = _load_dataset_chunks(dst_root, include_tables=True)
    _bridge_dataset_chunks(
        dst_chunks,
        gap_threshold=gap_threshold,
        indices=process_indices,
        target_values=process_target_values,
        inactive_value=inactive_value,
    )
    _write_chunk_actions(dst_chunks)

    if skip_recompute_stats:
        return

    dataset_cls, recompute_stats_fn = _require_lerobot()
    dataset = dataset_cls(repo_id=repo_id or dst_root.name, root=dst_root)
    recompute_stats_fn(dataset, skip_image_video=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and bridge short inactive gaps in LeRobot dataset action dimensions."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Source LeRobot dataset directory")
    parser.add_argument("--dst-root", type=Path, default=None, help="Destination dataset directory for rewritten data")
    parser.add_argument(
        "--analyze-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_ANALYZE_INDICES),
        help="Action indices to analyze before optionally rewriting the dataset",
    )
    parser.add_argument(
        "--process-indices",
        type=int,
        nargs="+",
        default=None,
        help="Action indices to rewrite in bridge mode",
    )
    parser.add_argument(
        "--target-values",
        type=float,
        nargs="+",
        default=None,
        help="Target values to bridge for each process index",
    )
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=None,
        help="Bridge inactive gaps with length less than or equal to this threshold",
    )
    parser.add_argument(
        "--inactive-value",
        type=float,
        default=DEFAULT_INACTIVE_VALUE,
        help="Value treated as an inactive gap",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Optional repo id for stats recomputation")
    parser.add_argument(
        "--skip-recompute-stats",
        action="store_true",
        help="Skip meta/stats.json recomputation after rewriting",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    analysis = analyze_dataset(
        args.src_root,
        analyze_indices=args.analyze_indices,
        inactive_value=args.inactive_value,
    )
    print(format_analysis_summary(analysis))

    if args.gap_threshold is None:
        return
    if args.dst_root is None:
        raise ValueError("--dst-root is required when --gap-threshold is provided.")
    if args.process_indices is None:
        raise ValueError("--process-indices is required when --gap-threshold is provided.")
    if args.target_values is None:
        raise ValueError("--target-values is required when --gap-threshold is provided.")

    process_dataset(
        args.src_root,
        args.dst_root,
        gap_threshold=args.gap_threshold,
        indices=args.process_indices,
        target_values=args.target_values,
        repo_id=args.repo_id,
        inactive_value=args.inactive_value,
        skip_recompute_stats=args.skip_recompute_stats,
    )


if __name__ == "__main__":
    main()
