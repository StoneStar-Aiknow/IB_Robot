"""Tests for frame_detector motion-current compatibility."""

from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.frame_detector import FrameDetector, FrameDetectorConfig  # noqa: E402


class _ListLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def debug(self, msg: str) -> None:
        self.messages.append(("debug", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))


def _make_detector(tmp_path: Path, *, features: dict, **overrides) -> tuple[FrameDetector, _ListLogger]:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": features,
            }
        ),
        encoding="utf-8",
    )

    cfg = FrameDetectorConfig(dataset_path=str(tmp_path))
    for key, value in overrides.items():
        setattr(cfg, key, value)

    logger = _ListLogger()
    return FrameDetector(cfg, logger=logger), logger


def _base_features(state_names: list[str]) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        }
    }


def test_log_uses_stable_call_sites_for_ros_logger() -> None:
    """Regression test for rclpy loggers rejecting changing severities per call site."""

    class StrictCallSiteLogger:
        def __init__(self):
            self.severity_by_call_site: dict[tuple[str, int], str] = {}

        def _record(self, severity: str, msg: str) -> None:
            frame = inspect.currentframe()
            assert frame is not None
            caller = frame.f_back.f_back
            call_site = (caller.f_code.co_filename, caller.f_lineno)
            previous = self.severity_by_call_site.setdefault(call_site, severity)
            if previous != severity:
                raise ValueError("Logger severity cannot be changed between calls.")

        def debug(self, msg: str) -> None:
            self._record("debug", msg)

        def info(self, msg: str) -> None:
            self._record("info", msg)

        def warning(self, msg: str) -> None:
            self._record("warning", msg)

        def error(self, msg: str) -> None:
            self._record("error", msg)

    detector = FrameDetector.__new__(FrameDetector)
    detector.logger = StrictCallSiteLogger()

    detector._log("info", "first")
    detector._log("warning", "second")
    detector._log("error", "third")


def test_analyze_skips_critical_detection_when_current_is_missing(tmp_path: Path):
    detector, logger = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "gripper.pos"]),
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
    )
    df = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5], [0.0, 0.5]],
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "index": [0, 1],
        }
    )

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [1.0, 1.0]
    assert "observation.current" not in analyzed
    assert any("critical-frame detection is skipped" in msg for level, msg in logger.messages if level == "warning")


def test_analyze_uses_velocity_only_freeze_detection_when_current_is_missing(tmp_path: Path):
    detector, logger = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "gripper.pos"]),
        enable_critical_detection=False,
        enable_freeze_detection=True,
        freeze_frame_min_duration=2,
    )
    df = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]],
            "timestamp": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "episode_index": [0, 0, 0],
            "index": [0, 1, 2],
        }
    )

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [0.0, 0.0, 0.0]
    assert any("using velocity only" in msg for level, msg in logger.messages if level == "warning")


def test_analyze_recovers_current_from_legacy_mixed_state_metadata(tmp_path: Path):
    detector, logger = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "gripper.pos", "joint.current", "gripper.current"]),
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=1,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )
    df = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5, 0.0, 0.7], [0.0, 0.5, 0.0, 0.8]],
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "index": [0, 1],
        }
    )

    analyzed = detector._analyze(df)

    assert analyzed["observation.state"].tolist() == [[0.0, 0.5], [0.0, 0.5]]
    assert analyzed["observation.current"].tolist() == [[0.0, 0.7], [0.0, 0.8]]
    assert analyzed["training_weight"].tolist() == [2.0, 2.0]
    assert detector.meta_info["features"]["observation.state"]["names"] == ["joint.pos", "gripper.pos"]
    assert detector.meta_info["features"]["observation.current"]["names"] == ["joint.current", "gripper.current"]
    assert detector._meta_info_dirty is True
    assert any("Recovered observation.current" in msg for level, msg in logger.messages if level == "warning")


def test_update_split_current_meta_defers_disk_write(tmp_path: Path, monkeypatch):
    detector, _logger = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "joint.current"]),
    )
    writes = []
    monkeypatch.setattr(detector, "_write_meta_info", lambda: writes.append("write"))

    detector._update_split_current_meta(["joint.pos"], ["joint.current"])

    assert writes == []
    assert detector._meta_info_dirty is True


def test_resolve_gripper_indices_single_arm_fallback(tmp_path: Path):
    names = ["position.1", "position.2", "position.3", "position.4", "position.5", "position.6"]
    detector, _ = _make_detector(
        tmp_path,
        features=_base_features(names),
        enable_critical_detection=True,
        gripper_pos=[-1],
    )
    assert detector.gripper_indices == [5]


def test_resolve_gripper_indices_dual_arm(tmp_path: Path):
    """Dual-arm names contain 'gripper' → name-based match."""
    names = [
        "position.leader_arm:shoulder_pan",
        "position.leader_arm:shoulder_lift",
        "position.leader_arm:elbow_flex",
        "position.leader_arm:wrist_flex",
        "position.leader_arm:wrist_roll",
        "position.leader_arm:gripper",
        "position.follower_arm:shoulder_pan",
        "position.follower_arm:shoulder_lift",
        "position.follower_arm:elbow_flex",
        "position.follower_arm:wrist_flex",
        "position.follower_arm:wrist_roll",
        "position.follower_arm:gripper",
    ]
    detector, _ = _make_detector(
        tmp_path,
        features=_base_features(names),
        enable_critical_detection=True,
        gripper_pos=[-1],
    )
    assert detector.gripper_indices == [5, 11]


def test_resolve_gripper_indices_fallback_no_names(tmp_path: Path):
    """No feature names in metadata → fallback to gripper_pos."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
        }
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=True,
        gripper_pos=[-1],
    )
    assert detector.gripper_indices == [5]


# ---------------------------------------------------------------------------
# Tests: critical_frame_min_duration
# ---------------------------------------------------------------------------

def test_critical_min_duration_filters_short_bursts(tmp_path: Path):
    """Single-frame critical spikes below min_duration should be discarded."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=3,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.0]] * 5,
        "observation.current": [[0.0, 0.6], [0.0, 0.6], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "timestamp": [i / 30.0 for i in range(5)],
        "episode_index": [0] * 5,
        "index": list(range(5)),
    })

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_critical_min_duration_keeps_long_bursts(tmp_path: Path):
    """Critical bursts >= min_duration frames should be kept."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=3,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.0]] * 5,
        "observation.current": [[0.0, 0.8], [0.0, 0.8], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0]],
        "timestamp": [i / 30.0 for i in range(5)],
        "episode_index": [0] * 5,
        "index": list(range(5)),
    })

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [2.0, 2.0, 2.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Tests: episode boundary isolation
# ---------------------------------------------------------------------------

def test_propagate_weights_respects_episode_boundary(tmp_path: Path):
    """Weight propagation must not cross episode boundaries."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=1,
        n_forward_expansion=3,
        n_backward_expansion=3,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.0]] * 6,
        "observation.current": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "timestamp": [i / 30.0 for i in range(6)],
        "episode_index": [0, 0, 0, 1, 1, 1],
        "index": list(range(6)),
    })

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]


def test_calculate_velocity_zero_at_episode_boundary():
    """Velocity must be zero at episode boundaries."""
    detector = FrameDetector.__new__(FrameDetector)
    state = pd.Series([[0.0, 0.5], [0.1, 0.6], [0.0, 0.0], [0.2, 0.3]])
    ts = pd.Series([0.0, 0.033, 0.066, 0.099])
    ep = pd.Series([0, 0, 1, 1])

    v = detector._calculate_velocity(state, ts, ep)

    assert v[0] == [0.0, 0.0]
    assert v[2] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Tests: all-zero current warning
# ---------------------------------------------------------------------------

def test_all_zero_current_emits_warning(tmp_path: Path):
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, logger = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=False,
        enable_freeze_detection=True,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.5]] * 5,
        "observation.current": [[0.0, 0.0]] * 5,
        "timestamp": [i / 30.0 for i in range(5)],
        "episode_index": [0] * 5,
        "index": list(range(5)),
    })

    detector._analyze(df)

    assert any("all zeros" in msg for _, msg in logger.messages)


def test_nonzero_current_does_not_warn(tmp_path: Path):
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, logger = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=False,
        enable_freeze_detection=True,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.5]] * 5,
        "observation.current": [[0.1, 0.2]] * 5,
        "timestamp": [i / 30.0 for i in range(5)],
        "episode_index": [0] * 5,
        "index": list(range(5)),
    })

    detector._analyze(df)

    assert not any("all zeros" in msg for _, msg in logger.messages)


# ---------------------------------------------------------------------------
# Tests: min_duration episode boundary isolation
# ---------------------------------------------------------------------------

def test_critical_min_duration_episode_isolation(tmp_path: Path):
    """Critical frames split across episode boundary must not merge for duration."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=3,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.0]] * 4,
        "observation.current": [[0.0, 0.8], [0.0, 0.8], [0.0, 0.8], [0.0, 0.8]],
        "timestamp": [i / 30.0 for i in range(4)],
        "episode_index": [0, 0, 1, 1],
        "index": list(range(4)),
    })

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_freeze_min_duration_episode_isolation(tmp_path: Path):
    """Static frames split across episode boundary must not merge for duration."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pos", "gripper"]},
        "observation.current": {"dtype": "float32", "shape": [2], "names": ["cur", "gripper_cur"]},
    }
    detector, _ = _make_detector(
        tmp_path,
        features=features,
        enable_critical_detection=False,
        enable_freeze_detection=True,
        freeze_frame_max_velocity=0.1,
        freeze_frame_max_current=0.1,
        freeze_frame_min_duration=5,
    )
    df = pd.DataFrame({
        "observation.state": [[0.0, 0.0]] * 8,
        "observation.current": [[0.0, 0.0]] * 8,
        "timestamp": [i / 30.0 for i in range(8)],
        "episode_index": [0, 0, 0, 0, 1, 1, 1, 1],
        "index": list(range(8)),
    })

    analyzed = detector._analyze(df)

    assert analyzed["training_weight"].tolist() == [1.0] * 8


# ---------------------------------------------------------------------------
# Tests: gripper_pos positive out-of-range
# ---------------------------------------------------------------------------

def test_gripper_pos_positive_out_of_range_raises(tmp_path: Path):
    """Positive gripper_pos >= state_dim should raise ValueError."""
    import pytest
    features = {
        "observation.state": {"dtype": "float32", "shape": [6]},
    }
    with pytest.raises(ValueError, match="out of range"):
        _make_detector(
            tmp_path,
            features=features,
            enable_critical_detection=True,
            gripper_pos=[7],
        )


def test_legacy_split_re_resolves_gripper_indices(tmp_path: Path):
    """After splitting mixed state/current, gripper_indices must target the split arrays.

    Without the fix, gripper_indices kept stale values (e.g. [1, 3] from 4-element
    mixed metadata) while the split arrays only have 2 elements each, causing
    _extract_data to silently pad zeros for out-of-range indices.
    """
    detector, _ = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "gripper.pos", "joint.current", "gripper.current"]),
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=1,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )

    assert detector.gripper_indices == [1, 3], "pre-split: both gripper entries matched"

    df = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5, 0.0, 0.7], [0.0, 0.5, 0.0, 0.8]],
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "index": [0, 1],
        }
    )

    detector._analyze(df)

    assert detector.gripper_indices == [1], (
        f"post-split: should target index 1 in 2-element arrays, got {detector.gripper_indices}"
    )


def test_legacy_split_multi_file_consistency(tmp_path: Path):
    """Second file must also split and detect critical frames.

    Without the fix, _update_split_current_meta mutated self.meta_info in-place,
    so the second file's _split_state_current_from_metadata read the updated
    metadata (no .current suffix) and returned None, losing critical detection.
    """
    detector, _ = _make_detector(
        tmp_path,
        features=_base_features(["joint.pos", "gripper.pos", "joint.current", "gripper.current"]),
        enable_critical_detection=True,
        enable_freeze_detection=False,
        gripper_pos=[-1],
        critical_frame_min_current_threshold=0.5,
        critical_frame_max_velocity_threshold=0.01,
        critical_frame_min_duration=1,
        n_forward_expansion=0,
        n_backward_expansion=0,
    )

    df1 = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5, 0.0, 0.7], [0.0, 0.5, 0.0, 0.8]],
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "index": [0, 1],
        }
    )
    df2 = pd.DataFrame(
        {
            "observation.state": [[0.0, 0.5, 0.0, 0.9], [0.0, 0.5, 0.0, 1.0]],
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "index": [0, 1],
        }
    )

    analyzed1 = detector._analyze(df1)
    analyzed2 = detector._analyze(df2)

    assert "observation.current" in analyzed2.columns, (
        "second file should still split out observation.current"
    )
    assert analyzed2["training_weight"].tolist() == [2.0, 2.0], (
        "second file should detect critical frames"
    )
