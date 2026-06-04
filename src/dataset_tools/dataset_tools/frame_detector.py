#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Detector for LeRobot Datasets - ROS 2 Node

Identifies critical and freeze frames in robot datasets and assigns training weights.

Usage:
    ros2 run dataset_tools frame_detector --ros-args -p dataset_path:=/path/to/dataset
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import traceback
import fnmatch
from dataclasses import dataclass, field
from typing import List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Warning: Some optional dependencies not available: {e}")


@dataclass
class FrameDetectorConfig:
    """Configuration for Frame Detector.
    
    Default values are optimized for banana grasping dataset.
    """
    
    dataset_path: str = ""
    clip_view: List[str] = field(default_factory=lambda: ["all"])
    enable_critical_detection: bool = True
    gripper_pos: List[int] = field(default_factory=lambda: [-1])
    gripper_names: List[str] = field(default_factory=lambda: ["*gripper*"])
    critical_frame_min_current_threshold: float = 0.5
    critical_frame_max_velocity_threshold: float = 0.01
    critical_frame_training_weight: float = 2.0
    critical_frame_min_duration: int = 3
    n_forward_expansion: int = 30
    n_backward_expansion: int = 30
    enable_freeze_detection: bool = True
    freeze_head_tail_only: bool = False
    freeze_frame_max_velocity: float = 0.1
    freeze_frame_max_current: float = 0.2
    freeze_frame_training_weight: float = 0.0
    freeze_frame_min_duration: int = 5
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.dataset_path:
            raise ValueError("dataset_path is required")
        if self.enable_critical_detection and not self.gripper_pos:
            raise ValueError("gripper_pos is required when enable_critical_detection is True")


def create_or_replace_directory(directory: str, force: bool = True, logger=None) -> bool:
    """Create or replace a directory.
    
    Args:
        directory: Path to directory
        force: If True, delete existing directory without asking
        logger: Optional ROS logger for logging
    """
    if os.path.exists(directory):
        if not force:
            if logger:
                logger.warning(f"Directory '{directory}' exists, operation cancelled")
            return False
        try:
            shutil.rmtree(directory)
            if logger:
                logger.info(f"Directory '{directory}' deleted")
        except Exception as e:
            if logger:
                logger.error(f"Error deleting directory: {e}")
            return False
    
    try:
        os.makedirs(directory, exist_ok=True)
        if logger:
            logger.info(f"Directory '{directory}' created")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error creating directory: {e}")
        return False


class FrameDetector:
    """Frame Detector for LeRobot datasets."""
    
    def __init__(self, cfg: FrameDetectorConfig, logger=None):
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path
        self.logger = logger
        
        self.total_files_processed = 0
        self.total_frames_analyzed = 0
        self.critical_frames_detected = 0
        self.freeze_frames_detected = 0
        self._warnings_emitted = set()
        self._meta_info_dirty = False
        
        meta_path = os.path.join(self.dataset_path, "meta", "info.json")
        with open(meta_path, encoding="utf-8") as f:
            self.meta_info = json.load(f)

        _state_feature = self.meta_info.get("features", {}).get("observation.state", {})
        self._original_state_names = list(_state_feature.get("names") or [])

        self.clip_view = cfg.clip_view
        
        self.enable_critical_detection = cfg.enable_critical_detection
        if self.enable_critical_detection:
            self.gripper_indices = self._resolve_gripper_indices()
            self.critical_frame_min_current_threshold = cfg.critical_frame_min_current_threshold
            self.critical_frame_max_velocity_threshold = cfg.critical_frame_max_velocity_threshold
            self.critical_frame_training_weight = cfg.critical_frame_training_weight
            self.critical_frame_min_duration = cfg.critical_frame_min_duration
            self.n_forward_expansion = cfg.n_forward_expansion
            self.n_backward_expansion = cfg.n_backward_expansion
        
        self.enable_freeze_detection = cfg.enable_freeze_detection
        if self.enable_freeze_detection:
            self.freeze_head_tail_only = cfg.freeze_head_tail_only
            self.freeze_frame_max_velocity = cfg.freeze_frame_max_velocity
            self.freeze_frame_max_current = cfg.freeze_frame_max_current
            self.freeze_frame_training_weight = cfg.freeze_frame_training_weight
            self.freeze_frame_min_duration = cfg.freeze_frame_min_duration
    
    def _log(self, level: str, msg: str):
        """Log message using ROS logger or print."""
        if self.logger:
            if level == "debug":
                self.logger.debug(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
            else:
                self.logger.info(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def _resolve_gripper_indices(self) -> list[int]:
        """Resolve gripper indices from feature names, falling back to config.

        Negative indices in *gripper_pos* are resolved against the
        ``observation.state`` feature dimension (e.g. ``-1`` with shape
        ``[6]`` → ``5``).
        """
        state_info = self.meta_info.get("features", {}).get("observation.state", {})
        names = state_info.get("names", [])
        state_dim = state_info.get("shape", [0])[0] if state_info.get("shape") else 0

        def _normalise(raw: list[int]) -> list[int]:
            """Resolve negative indices using state_dim; reject positive out-of-range."""
            result = []
            for i in raw:
                if i < 0:
                    if state_dim > 0:
                        i = i % state_dim
                elif state_dim > 0 and i >= state_dim:
                    raise ValueError(f"gripper_pos index {i} out of range [0, {state_dim})")
                result.append(i)
            return result

        if not names:
            return _normalise(self.cfg.gripper_pos)

        indices: list[int] = []
        for pattern in self.cfg.gripper_names:
            for i, name in enumerate(names):
                if fnmatch.fnmatch(str(name).lower(), f"*{pattern.lower()}*"):
                    if i not in indices:
                        indices.append(i)

        return indices if indices else _normalise(self.cfg.gripper_pos)
    
    def run(self):
        """Main execution flow."""
        if not self.enable_critical_detection and not self.enable_freeze_detection:
            self._log("warning", "No detection enabled, exiting")
            return False
        
        data_dir = os.path.join(self.dataset_path, "data")
        files = []
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.parquet'):
                    files.append(os.path.join(root, filename))
        files.sort()
        total_files = len(files)
        
        if total_files == 0:
            self._log("error", "No parquet files found")
            return False
        
        view_dirs = self._get_clip_view()
        self._add_training_weight_feature_to_meta_info()
        self._write_meta_info()
        
        clip_dir = os.path.join(self.dataset_path, "video_clips")
        if not create_or_replace_directory(clip_dir, force=True, logger=self.logger):
            return False
        
        all_processed_dfs = []
        
        for idx, file in enumerate(files):
            self._log("info", f"Processing file {idx + 1}/{total_files}: {file}")
            
            df = pd.read_parquet(file)
            df = self._analyze(df)
            all_processed_dfs.append(df)
            
            self.total_files_processed += 1
            self.total_frames_analyzed += len(df)
            self.critical_frames_detected += len(df[df["training_weight"] == self.critical_frame_training_weight]) if self.enable_critical_detection else 0
            self.freeze_frames_detected += len(df[df["training_weight"] == self.freeze_frame_training_weight]) if self.enable_freeze_detection else 0
            
            file_name = os.path.basename(file).replace('.parquet', '')
            for view in view_dirs:
                clip_output_path = os.path.join(clip_dir, view, file_name)
                video_path = os.path.join(self.dataset_path, "videos", view, f"{file_name}.mp4")
                if os.path.exists(video_path):
                    commands = self._generate_video_clip_commands(df, video_path, clip_output_path)
                    for command in commands:
                        subprocess.run(command, check=True, capture_output=True, text=True)
            
            self._write_parquet(df, file)

        if self._meta_info_dirty:
            self._write_meta_info()
        
        if all_processed_dfs:
            self._log("info", "Generating global distribution plot...")
            full_df = pd.concat(all_processed_dfs, ignore_index=True)
            self._visualize_weight(full_df, max_episodes=10)
        
        return True
    
    def _analyze(self, df):
        """Execute complete analysis."""
        df["training_weight"] = 1.0
        state_series, current_series = self._resolve_motion_series(df)

        if current_series is not None:
            self._warn_all_zero_current(current_series)

        # Critical frame detection
        if self.enable_critical_detection:
            if current_series is None:
                self._warn_once(
                    "missing_current_critical",
                    "Dataset does not contain observation.current; "
                    "critical-frame detection is skipped for this dataset.",
                )
            else:
                df["gripper_current"] = self._extract_data(current_series, indices=self.gripper_indices)
                df["gripper_state"] = self._extract_data(state_series, indices=self.gripper_indices)
                df["gripper_velocity"] = self._calculate_velocity(
                    df["gripper_state"], df["timestamp"], df.get("episode_index")
                )
                df["training_weight"] = df.apply(self._calculate_critical_weight, axis=1)

                if self.critical_frame_min_duration > 1:
                    critical_mask = df["training_weight"] == self.critical_frame_training_weight
                    groups = self._episode_aware_groups(critical_mask, df)
                    group_sizes = critical_mask.groupby(groups).transform("size")
                    df.loc[critical_mask & (group_sizes < self.critical_frame_min_duration), "training_weight"] = 1.0

                df["training_weight"] = self._propagate_weights(df)
        
        # Freeze frame detection
        if self.enable_freeze_detection:
            use_current = current_series is not None
            if use_current:
                df["all_current"] = self._extract_data(current_series, indices=None)
            else:
                self._warn_once(
                    "missing_current_freeze",
                    "Dataset does not contain observation.current; "
                    "freeze-frame detection is using velocity only.",
                )
            df["all_state"] = self._extract_data(state_series, indices=None)
            df["all_velocity"] = self._calculate_velocity(
                df["all_state"], df["timestamp"], df.get("episode_index")
            )
            df = self._detect_freeze_frames(df, use_current=use_current)
        
        # Cleanup columns
        if self.enable_critical_detection:
            df = df.drop(["gripper_current", "gripper_state", "gripper_velocity"], axis=1, errors='ignore')
        if self.enable_freeze_detection:
            df = df.drop(["all_current", "all_state", "all_velocity"], axis=1, errors='ignore')
        
        return df

    def _warn_once(self, key: str, msg: str) -> None:
        """Emit a warning at most once per detector run."""
        if key in self._warnings_emitted:
            return
        self._warnings_emitted.add(key)
        self._log("warning", msg)

    def _warn_all_zero_current(self, current_series) -> None:
        """Warn if observation.current is entirely zero (likely name mismatch)."""
        sample_size = min(100, len(current_series))
        for row in current_series.head(sample_size):
            if hasattr(row, "__iter__"):
                if any(float(v) != 0.0 for v in row):
                    return
            elif float(row) != 0.0:
                return
        self._warn_once(
            "all_zero_current",
            f"observation.current is all zeros in the first {sample_size} frames. "
            "This may indicate a joint-name mismatch between the YAML selector "
            "and the configured current topic. Verify selector names (e.g. 'current.1') match "
            "the joint names published by the hardware.",
        )

    def _resolve_motion_series(self, df):
        """Return position and current series, recovering legacy mixed-state datasets when possible."""
        if "observation.state" not in df:
            raise KeyError("observation.state")

        state_series = df["observation.state"]
        if "observation.current" in df:
            return state_series, df["observation.current"]

        split = self._split_state_current_from_metadata(state_series)
        if split is None:
            return state_series, None

        state_series, current_series, state_names, current_names = split
        df["observation.state"] = state_series
        df["observation.current"] = current_series
        self._update_split_current_meta(state_names, current_names)
        if self.enable_critical_detection:
            self.gripper_indices = self._resolve_gripper_indices()
        self._warn_once(
            "split_mixed_state_current",
            "Recovered observation.current from observation.state metadata; "
            "rewriting observation.state as position-only values.",
        )
        return state_series, current_series

    def _split_state_current_from_metadata(self, state_series):
        """Split legacy state vectors that contain both *.pos and *.current values."""
        names = self._original_state_names
        if not names:
            return None

        current_indices = [idx for idx, name in enumerate(names) if str(name).endswith(".current")]
        if not current_indices:
            return None

        state_indices = [idx for idx, name in enumerate(names) if idx not in current_indices]
        state_names = [names[idx] for idx in state_indices]
        current_names = [names[idx] for idx in current_indices]

        position_values = self._extract_data(state_series, indices=state_indices)
        current_values = self._extract_data(state_series, indices=current_indices)
        return (
            pd.Series(position_values, index=state_series.index),
            pd.Series(current_values, index=state_series.index),
            state_names,
            current_names,
        )

    def _update_split_current_meta(self, state_names, current_names) -> None:
        """Update dataset metadata after splitting legacy mixed state/current vectors."""
        features = self.meta_info.setdefault("features", {})
        state_feature = features.setdefault("observation.state", {})
        state_feature["dtype"] = "float32"
        state_feature["shape"] = [len(state_names)]
        state_feature["names"] = state_names
        features["observation.current"] = {
            "dtype": "float32",
            "shape": [len(current_names)],
            "names": current_names,
        }
        self._meta_info_dirty = True
    
    def _extract_data(self, series, indices=None):
        """Extract motor data from series."""
        result = []
        for x in series:
            if hasattr(x, "__getitem__"):
                if indices is None:
                    result.append(list(x))
                else:
                    data = []
                    for idx in indices:
                        if 0 <= idx < len(x) or -len(x) <= idx < 0:
                            data.append(x[idx])
                        else:
                            data.append(0)
                    result.append(data)
            else:
                length = len(indices) if indices is not None else 1
                result.append([0] * length)
        return result
    
    def _calculate_velocity(self, state_series, timestamp_series, episode_series=None):
        """Calculate velocity from state series."""
        ep_values = episode_series.values if episode_series is not None else None
        velocities = []
        for i in range(len(state_series)):
            if i == 0:
                num_joints = len(state_series.iloc[i]) if hasattr(state_series.iloc[i], "__len__") else 1
                velocities.append([0.0] * num_joints)
            elif ep_values is not None and ep_values[i] != ep_values[i - 1]:
                num_joints = len(state_series.iloc[i]) if hasattr(state_series.iloc[i], "__len__") else 1
                velocities.append([0.0] * num_joints)
            else:
                dt = timestamp_series.iloc[i] - timestamp_series.iloc[i - 1]
                current_state = state_series.iloc[i]
                prev_state = state_series.iloc[i - 1]
                
                if hasattr(current_state, "__len__") and hasattr(prev_state, "__len__") and len(current_state) == len(prev_state):
                    joint_velocities = []
                    for j in range(len(current_state)):
                        v = (current_state[j] - prev_state[j]) / dt if dt > 0 else 0.0
                        joint_velocities.append(v)
                    velocities.append(joint_velocities)
                else:
                    num_joints = len(current_state) if hasattr(current_state, "__len__") else 1
                    velocities.append([0.0] * num_joints)
        return velocities
    
    def _episode_aware_groups(self, mask, df):
        """Group consecutive identical mask values, respecting episode boundaries."""
        status_change = mask != mask.shift()
        ep = df.get("episode_index")
        if ep is not None:
            return (status_change | (ep != ep.shift())).cumsum()
        return status_change.cumsum()

    def _calculate_critical_weight(self, row):
        """Calculate training weight for critical frames."""
        for i in range(len(row["gripper_velocity"])):
            velocity = row["gripper_velocity"][i]
            current = row["gripper_current"][i]
            if abs(velocity) <= self.critical_frame_max_velocity_threshold and abs(current) >= self.critical_frame_min_current_threshold:
                return self.critical_frame_training_weight
        return 1.0
    
    def _propagate_weights(self, df):
        """Propagate weights to neighboring frames."""
        new_weights = df["training_weight"].copy()
        weight_indices = df[df["training_weight"] == self.critical_frame_training_weight].index

        ep_values = df["episode_index"].values if "episode_index" in df.columns else None

        for idx in weight_indices:
            ep = ep_values[idx] if ep_values is not None else None

            # Backward expansion — propagate to preceding frames
            start = max(0, idx - self.n_backward_expansion)
            for i in range(idx - 1, start - 1, -1):
                if ep_values is not None and ep_values[i] != ep:
                    break
                new_weights.iloc[i] = self.critical_frame_training_weight

            # Forward expansion — propagate to following frames
            end = min(len(df) - 1, idx + self.n_forward_expansion)
            for i in range(idx + 1, end + 1):
                if ep_values is not None and ep_values[i] != ep:
                    break
                new_weights.iloc[i] = self.critical_frame_training_weight

        return new_weights
    
    def _detect_freeze_frames(self, df, use_current: bool = True):
        """Detect freeze frames based on velocity and current thresholds."""
        
        def is_row_instant_static(row):
            velocities = row["all_velocity"]
            v_static = all(abs(v) <= self.freeze_frame_max_velocity for v in velocities)
            if use_current:
                currents = row["all_current"]
                c_static = all(abs(c) <= self.freeze_frame_max_current for c in currents)
            else:
                c_static = True
            return v_static and c_static
        
        # Calculate static mask
        raw_static_mask = df.apply(is_row_instant_static, axis=1)
        
        # Filter by duration
        if self.freeze_frame_min_duration > 1:
            groups = self._episode_aware_groups(raw_static_mask, df)
            group_sizes = raw_static_mask.groupby(groups).transform("size")
            valid_static_mask = raw_static_mask & (group_sizes >= self.freeze_frame_min_duration)
        else:
            valid_static_mask = raw_static_mask
        
        # Apply weights
        if self.freeze_head_tail_only:
            for episode_idx in df["episode_index"].unique():
                episode_mask = df["episode_index"] == episode_idx
                indices = df[episode_mask].index
                if len(indices) == 0:
                    continue
                
                # Head
                for idx in indices:
                    if valid_static_mask[idx]:
                        df.at[idx, "training_weight"] = self.freeze_frame_training_weight
                    else:
                        break
                
                # Tail
                for idx in reversed(indices):
                    if valid_static_mask[idx]:
                        df.at[idx, "training_weight"] = self.freeze_frame_training_weight
                    else:
                        break
        else:
            df.loc[valid_static_mask, "training_weight"] = self.freeze_frame_training_weight
        
        return df
    
    def _find_weight_intervals(self, df):
        """Find all intervals where weight is not 1."""
        intervals = []
        in_interval = False
        start_time = 0
        start_index = 0
        
        start_time_map = self._find_start_time(df)
        df_reset = df.reset_index(drop=True)
        
        for i, row in df_reset.iterrows():
            current_weight = row["training_weight"]
            ep_idx = int(row["episode_index"])
            
            if current_weight != 1 and not in_interval:
                in_interval = True
                start_time = row["timestamp"] + start_time_map[ep_idx]
                start_index = int(row["index"])
            
            elif in_interval:
                prev_row = df_reset.iloc[i - 1]
                prev_ep_idx = int(prev_row["episode_index"])
                
                if current_weight == 1 or ep_idx != prev_ep_idx:
                    in_interval = False
                    end_time = prev_row["timestamp"] + start_time_map[prev_ep_idx]
                    end_index = int(prev_row["index"])
                    weight_type = prev_row["training_weight"]
                    intervals.append(((start_time, end_time), (start_index, end_index), prev_ep_idx, weight_type))
                    
                    if current_weight != 1 and ep_idx != prev_ep_idx:
                        in_interval = True
                        start_time = row["timestamp"] + start_time_map[ep_idx]
                        start_index = int(row["index"])
        
        if in_interval:
            last_row = df_reset.iloc[-1]
            last_ep_idx = int(last_row["episode_index"])
            end_time = last_row["timestamp"] + start_time_map[last_ep_idx]
            end_index = int(last_row["index"])
            weight_type = last_row["training_weight"]
            intervals.append(((start_time, end_time), (end_index, end_index), last_ep_idx, weight_type))
        
        return intervals
    
    def _find_start_time(self, df):
        """Calculate start time for each episode."""
        unique_episodes = sorted(df["episode_index"].unique())
        group_durations = df.groupby("episode_index")["timestamp"].max() + 1 / self.meta_info.get("fps", 30)
        
        start_times = {}
        current_cumulative_time = 0.0
        
        for ep_idx in unique_episodes:
            start_times[ep_idx] = current_cumulative_time
            current_cumulative_time += group_durations[ep_idx]
        
        return start_times
    
    def _generate_video_clip_commands(self, df, video_file, output_dir):
        """Generate ffmpeg commands for video clipping."""
        os.makedirs(output_dir, exist_ok=True)
        weight_intervals = self._find_weight_intervals(df)
        commands = []
        
        for i, ((start_time, end_time), _, episode_index, weight_val) in enumerate(weight_intervals):
            duration = end_time - start_time
            
            if self.enable_freeze_detection and weight_val == self.freeze_frame_training_weight:
                type_str = "FREEZE"
            elif self.enable_critical_detection and weight_val == self.critical_frame_training_weight:
                type_str = "CRITICAL"
            else:
                type_str = "OTHER"
            
            output_file = os.path.join(output_dir, f"{type_str}_clip_{i+1:03d}_ep{episode_index}_({start_time:.2f}-{end_time:.2f}).mp4")
            
            if duration > 0:
                cmd = [
                    "ffmpeg", "-y", "-i", video_file,
                    "-ss", str(start_time), "-t", str(duration),
                    "-c", "copy", output_file, "-loglevel", "error"
                ]
                commands.append(cmd)
        
        return commands
    
    def _get_clip_view(self):
        """Get clip view directories."""
        all_view_names = []
        all_view_dir = []
        view_dirs = []
        
        for i in self.meta_info.get("features", []):
            if "observation.images" in i:
                all_view_dir.append(i)
                all_view_names.append(i.split(".")[-1])
        
        if "all" not in self.clip_view:
            for i in self.clip_view:
                if i not in all_view_names:
                    self._log("warning", f"Dataset doesn't have {i} view, skipping.")
                else:
                    view_dirs.append(f"observation.images.{i}")
        else:
            self.clip_view = all_view_names
            view_dirs = all_view_dir
        
        return view_dirs
    
    def _add_training_weight_feature_to_meta_info(self):
        """Add the training_weight feature definition to meta info."""
        self.meta_info["features"]["training_weight"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }

    def _write_meta_info(self):
        """Write updated meta info to disk."""
        meta_path = os.path.join(self.dataset_path, "meta", "info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_info, f, ensure_ascii=False, indent=4)
    
    def _write_parquet(self, df, path):
        """Write DataFrame to parquet."""
        df.to_parquet(path, index=False)
    
    def _visualize_weight(self, df, max_episodes=None):
        """Visualize training weight distribution."""
        if not plt:
            self._log("warning", "matplotlib not available, skipping visualization")
            return
        
        all_episodes = sorted(df["episode_index"].unique())
        if max_episodes is not None and max_episodes < len(all_episodes):
            allowed_episodes = all_episodes[:max_episodes]
            df = df[df["episode_index"].isin(allowed_episodes)].copy()
            self._log("info", f"Visualization limited to first {max_episodes} episodes")
        
        if df.empty:
            return
        
        plt.figure(figsize=(15, 6))
        df = df.sort_values("index")
        
        # Plot main weight curve
        plt.plot(df["index"], df["training_weight"], "o-", markersize=1, linewidth=0.1, color="black")
        
        # Find and highlight weight intervals
        weight_intervals = self._find_weight_intervals(df)
        added_labels = set()
        
        for _, ((_, _), (start_idx, end_idx), _, weight_val) in enumerate(weight_intervals):
            if self.enable_critical_detection and weight_val == self.critical_frame_training_weight:
                color, label = "red", "Critical Frame"
            elif self.enable_freeze_detection and weight_val == self.freeze_frame_training_weight:
                color, label = "blue", "Freeze Frame"
            else:
                color, label = "gray", "Other"

            if label not in added_labels:
                plt.axvspan(start_idx, end_idx, alpha=0.3, color=color, label=label)
                added_labels.add(label)
            else:
                plt.axvspan(start_idx, end_idx, alpha=0.3, color=color)
        
        # Add episode separators
        unique_episodes = sorted(df["episode_index"].unique())
        y_min, y_max = plt.ylim()
        y_limit_top = 2.2 if y_max < 2 else y_max * 1.1
        plt.ylim(0, y_limit_top)
        
        for i, ep in enumerate(unique_episodes):
            ep_data = df[df["episode_index"] == ep]
            ep_start_idx = ep_data["index"].min()
            ep_end_idx = ep_data["index"].max()
            
            mid_point = (ep_start_idx + ep_end_idx) / 2
            plt.text(mid_point, y_limit_top * 0.95, f"Ep{ep}",
                    ha="center", va="top", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7})
            
            if i < len(unique_episodes) - 1:
                plt.axvline(x=ep_end_idx + 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        
        plt.xlabel("Global Index")
        plt.ylabel("Training Weight")
        plt.title(f"Training Weight Distribution (Episodes: {unique_episodes[0]}-{unique_episodes[-1]})")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.2)
        
        png_path = os.path.join(self.dataset_path, "training_weights_distribution.png")
        plt.savefig(png_path, dpi=200)
        self._log("info", f"Training weights distribution saved to {png_path}")
        plt.close()


class FrameDetectorNode(Node):
    """ROS 2 Node for Frame Detection.
    
    Reads dataset path from ROS parameter and executes frame detection.
    """
    
    def __init__(self):
        super().__init__("frame_detector")
        
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("clip_view", ["all"])
        self.declare_parameter("enable_critical_detection", True)
        self.declare_parameter("enable_freeze_detection", True)
        self.declare_parameter("gripper_pos", [-1])
        self.declare_parameter("critical_frame_min_current_threshold", 0.5)
        self.declare_parameter("critical_frame_max_velocity_threshold", 0.01)
        self.declare_parameter("critical_frame_training_weight", 2.0)
        self.declare_parameter("n_forward_expansion", 30)
        self.declare_parameter("n_backward_expansion", 30)
        self.declare_parameter("freeze_head_tail_only", False)
        self.declare_parameter("freeze_frame_max_velocity", 0.1)
        self.declare_parameter("freeze_frame_max_current", 0.2)
        self.declare_parameter("freeze_frame_training_weight", 0.0)
        self.declare_parameter("freeze_frame_min_duration", 5)
        self.declare_parameter("gripper_names", ["*gripper*"])
        self.declare_parameter("critical_frame_min_duration", 3)
    
    def run_detection(self) -> bool:
        """Execute frame detection with parameters."""
        cfg = FrameDetectorConfig()
        cfg.dataset_path = self.get_parameter("dataset_path").get_parameter_value().string_value
        cfg.clip_view = list(self.get_parameter("clip_view").get_parameter_value().string_array_value)
        cfg.enable_critical_detection = self.get_parameter("enable_critical_detection").get_parameter_value().bool_value
        cfg.enable_freeze_detection = self.get_parameter("enable_freeze_detection").get_parameter_value().bool_value
        cfg.gripper_pos = list(self.get_parameter("gripper_pos").get_parameter_value().integer_array_value)
        cfg.critical_frame_min_current_threshold = self.get_parameter("critical_frame_min_current_threshold").get_parameter_value().double_value
        cfg.critical_frame_max_velocity_threshold = self.get_parameter("critical_frame_max_velocity_threshold").get_parameter_value().double_value
        cfg.critical_frame_training_weight = self.get_parameter("critical_frame_training_weight").get_parameter_value().double_value
        cfg.n_forward_expansion = self.get_parameter("n_forward_expansion").get_parameter_value().integer_value
        cfg.n_backward_expansion = self.get_parameter("n_backward_expansion").get_parameter_value().integer_value
        cfg.freeze_head_tail_only = self.get_parameter("freeze_head_tail_only").get_parameter_value().bool_value
        cfg.freeze_frame_max_velocity = self.get_parameter("freeze_frame_max_velocity").get_parameter_value().double_value
        cfg.freeze_frame_max_current = self.get_parameter("freeze_frame_max_current").get_parameter_value().double_value
        cfg.freeze_frame_training_weight = self.get_parameter("freeze_frame_training_weight").get_parameter_value().double_value
        cfg.freeze_frame_min_duration = self.get_parameter("freeze_frame_min_duration").get_parameter_value().integer_value
        cfg.gripper_names = list(self.get_parameter("gripper_names").get_parameter_value().string_array_value)
        cfg.critical_frame_min_duration = self.get_parameter("critical_frame_min_duration").get_parameter_value().integer_value
        
        try:
            cfg.validate()
        except ValueError as e:
            self.get_logger().error(f"Configuration error: {e}")
            return False
        
        detector = FrameDetector(cfg, logger=self.get_logger())
        success = detector.run()
        
        if success:
            self.get_logger().info(
                f"Frame detection completed: {detector.total_files_processed} files, "
                f"{detector.total_frames_analyzed} frames, "
                f"{detector.critical_frames_detected} critical, "
                f"{detector.freeze_frames_detected} freeze"
            )
        return success

def main(args=None):
    """ROS 2 entry point."""
    rclpy.init(args=args)
    
    try:
        node = FrameDetectorNode()
        success = node.run_detection()
        node.destroy_node()
    except ExternalShutdownException:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
