from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools import lerobot_action_gap_repair as module  # noqa: E402


def _write_actions_parquet(parquet_path: Path, actions: list[list[float]], episode_index: list[int]) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    width = len(actions[0]) if actions else 0
    table = pa.Table.from_arrays(
        [
            pa.array(actions, type=pa.list_(pa.float32(), width)),
            pa.array(episode_index, type=pa.int64()),
        ],
        names=["action", "episode_index"],
    )
    pq.write_table(table, parquet_path)


def _read_actions(parquet_path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(parquet_path)["action"].to_pylist(), dtype=np.float32)


def test_analyze_action_parquet_file_reports_value_and_gap_distributions(tmp_path: Path) -> None:
    parquet_path = tmp_path / "actions.parquet"
    _write_actions_parquet(
        parquet_path,
        actions=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        ],
        episode_index=[0, 0, 0],
    )

    analysis = module.analyze_action_parquet_file(
        parquet_path,
        analyze_indices=[6, 8],
        inactive_value=0.0,
    )

    assert analysis[6]["value_counts"] == {0.0: 1, 0.1: 2}
    assert analysis[8]["gap_counts"][30.0] == {1: 1}


def test_analyze_dataset_merges_gap_counts_across_parquet_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    _write_actions_parquet(
        data_dir / "file-000.parquet",
        actions=[[0.0] * 8 + [30.0], [0.0] * 9, [0.0] * 8 + [30.0]],
        episode_index=[0, 0, 0],
    )
    _write_actions_parquet(
        data_dir / "file-001.parquet",
        actions=[[0.0] * 8 + [30.0], [0.0] * 9, [0.0] * 8 + [30.0]],
        episode_index=[1, 1, 1],
    )

    analysis = module.analyze_dataset(tmp_path, analyze_indices=[8], inactive_value=0.0)

    assert analysis[8]["value_counts"][30.0] == 4
    assert analysis[8]["gap_counts"][30.0] == {1: 2}


def test_analyze_dataset_tracks_gaps_across_parquet_boundaries_within_episode(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    _write_actions_parquet(
        data_dir / "file-000.parquet",
        actions=[[0.0] * 8 + [30.0], [0.0] * 9],
        episode_index=[0, 0],
    )
    _write_actions_parquet(
        data_dir / "file-001.parquet",
        actions=[[0.0] * 8 + [30.0]],
        episode_index=[0],
    )

    analysis = module.analyze_dataset(tmp_path, analyze_indices=[8], inactive_value=0.0)

    assert analysis[8]["value_counts"][30.0] == 2
    assert analysis[8]["gap_counts"][30.0] == {1: 1}


def test_bridge_target_value_gaps_only_bridges_inactive_short_gaps() -> None:
    values = np.array([0.1, 0.0, 0.1, 0.2, 0.1], dtype=np.float32)

    bridged = module.bridge_target_value_gaps(
        values,
        gap_threshold=1,
        target_value=0.1,
        inactive_value=0.0,
    )

    np.testing.assert_array_equal(bridged, np.array([0.1, 0.1, 0.1, 0.2, 0.1], dtype=np.float32))


def test_rewrite_action_parquet_file_updates_multiple_control_indices(tmp_path: Path) -> None:
    parquet_path = tmp_path / "actions.parquet"
    _write_actions_parquet(
        parquet_path,
        actions=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 9.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 9.0, 30.0],
        ],
        episode_index=[0, 0, 0],
    )

    module.rewrite_action_parquet_file(
        parquet_path,
        gap_threshold=1,
        indices=[6, 8],
        target_values=[0.1, 30.0],
    )

    rewritten_actions = _read_actions(parquet_path)
    np.testing.assert_array_equal(rewritten_actions[:, 6], np.array([0.1, 0.1, 0.1], dtype=np.float32))
    np.testing.assert_array_equal(rewritten_actions[:, 8], np.array([30.0, 30.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(rewritten_actions[:, 7], np.array([9.0, 9.0, 9.0], dtype=np.float32))


def test_rewrite_action_parquet_file_rejects_mismatched_index_and_target_lengths(tmp_path: Path) -> None:
    parquet_path = tmp_path / "actions.parquet"
    _write_actions_parquet(parquet_path, actions=[[0.0] * 9], episode_index=[0])

    with pytest.raises(ValueError, match="same length"):
        module.rewrite_action_parquet_file(
            parquet_path,
            gap_threshold=1,
            indices=[6, 8],
            target_values=[0.1],
        )


def test_rewrite_action_parquet_file_rejects_out_of_range_action_index(tmp_path: Path) -> None:
    parquet_path = tmp_path / "actions.parquet"
    _write_actions_parquet(parquet_path, actions=[[0.0] * 9], episode_index=[0])

    with pytest.raises(IndexError, match="out of range"):
        module.rewrite_action_parquet_file(
            parquet_path,
            gap_threshold=1,
            indices=[9],
            target_values=[0.1],
        )


def test_process_dataset_copies_dataset_and_rewrites_target_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_root = tmp_path / "src_ds"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        ],
        episode_index=[0, 0, 0],
    )

    calls: dict[str, object] = {}

    class DummyDataset:
        def __init__(self, repo_id: str, root: Path):
            calls["repo_id"] = repo_id
            calls["root"] = root

    def fake_recompute_stats(dataset: object, skip_image_video: bool = True) -> None:
        calls["recompute"] = (dataset, skip_image_video)

    monkeypatch.setattr(module, "LeRobotDataset", DummyDataset)
    monkeypatch.setattr(module, "recompute_stats", fake_recompute_stats)

    dst_root = tmp_path / "dst_ds"
    module.process_dataset(
        src_root,
        dst_root,
        gap_threshold=1,
        indices=[6, 8],
        target_values=[0.1, 30.0],
    )

    rewritten = _read_actions(dst_root / "data" / "chunk-000" / "file-000.parquet")
    np.testing.assert_array_equal(rewritten[:, 6], np.array([0.1, 0.1, 0.1], dtype=np.float32))
    np.testing.assert_array_equal(rewritten[:, 8], np.array([30.0, 30.0, 30.0], dtype=np.float32))
    assert calls["root"] == dst_root
    assert calls["repo_id"] == dst_root.name
    assert calls["recompute"][1] is True


def test_process_dataset_bridges_single_episode_across_parquet_boundaries(tmp_path: Path) -> None:
    src_root = tmp_path / "src_ds"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[[0.0] * 8 + [30.0], [0.0] * 9],
        episode_index=[0, 0],
    )
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-001.parquet",
        actions=[[0.0] * 8 + [30.0]],
        episode_index=[0],
    )

    dst_root = tmp_path / "dst_ds"
    module.process_dataset(
        src_root,
        dst_root,
        gap_threshold=1,
        indices=[8],
        target_values=[30.0],
        skip_recompute_stats=True,
    )

    first_chunk = _read_actions(dst_root / "data" / "chunk-000" / "file-000.parquet")
    second_chunk = _read_actions(dst_root / "data" / "chunk-000" / "file-001.parquet")
    np.testing.assert_array_equal(first_chunk[:, 8], np.array([30.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(second_chunk[:, 8], np.array([30.0], dtype=np.float32))


def test_process_dataset_validates_before_copying_destination(tmp_path: Path) -> None:
    src_root = tmp_path / "src_ds"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[[0.0] * 9],
        episode_index=[0],
    )
    dst_root = tmp_path / "dst_ds"

    with pytest.raises(ValueError, match="same length"):
        module.process_dataset(
            src_root,
            dst_root,
            gap_threshold=1,
            indices=[6, 8],
            target_values=[0.1],
            skip_recompute_stats=True,
        )

    assert not dst_root.exists()


def test_main_analysis_only_mode_prints_summary_and_does_not_rewrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src_root = tmp_path / "dataset"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0]],
        episode_index=[0],
    )

    module.main(["--src-root", str(src_root)])

    captured = capsys.readouterr()
    assert "action[8] value counts:" in captured.out
    assert not (tmp_path / "dataset_gap").exists()


def test_main_requires_dst_root_when_gap_threshold_is_provided(tmp_path: Path) -> None:
    src_root = tmp_path / "dataset"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[[0.0] * 9],
        episode_index=[0],
    )

    with pytest.raises(ValueError, match="dst-root"):
        module.main(["--src-root", str(src_root), "--gap-threshold", "1"])


def test_main_rewrite_mode_creates_new_dataset(tmp_path: Path) -> None:
    src_root = tmp_path / "dataset"
    _write_actions_parquet(
        src_root / "data" / "chunk-000" / "file-000.parquet",
        actions=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        ],
        episode_index=[0, 0, 0],
    )
    dst_root = tmp_path / "dataset_gap"

    module.main(
        [
            "--src-root",
            str(src_root),
            "--dst-root",
            str(dst_root),
            "--gap-threshold",
            "1",
            "--process-indices",
            "6",
            "8",
            "--target-values",
            "0.1",
            "30",
            "--skip-recompute-stats",
        ]
    )

    rewritten = _read_actions(dst_root / "data" / "chunk-000" / "file-000.parquet")
    np.testing.assert_array_equal(rewritten[:, 6], np.array([0.1, 0.1, 0.1], dtype=np.float32))
    np.testing.assert_array_equal(rewritten[:, 8], np.array([30.0, 30.0, 30.0], dtype=np.float32))


def test_setup_py_registers_lerobot_action_gap_repair_entrypoint() -> None:
    setup_py = (Path(__file__).resolve().parent.parent / "setup.py").read_text(encoding="utf-8")
    assert "lerobot_action_gap_repair = dataset_tools.lerobot_action_gap_repair:main" in setup_py
