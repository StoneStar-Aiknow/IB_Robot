# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A fixed follower arm calibration for lekiwi_robot tests that render the physical robot.

A physical render reads the arm calibration from ``$(env HOME)/.calibrate``. A
test that renders the physical robot requests ``pinned_calibration`` so it
reads this fixed file, never the calibration of an arm attached to the test
machine (issue #248). The fixture is not autouse, so the Gazebo and MuJoCo
integration tests keep their caches and ROS logs under the real HOME; the
simulated cases of a test that requests it only project the model and start
no simulator.

TEST DATA ONLY, never deploy it: the SO-101 follower ranges also used by
so101_robot's launch tests, homing_offset zeroed (the joint conversion reads
only the ranges). Wheels 7-9 are velocity joints and have no entry.
"""

from __future__ import annotations

import copy
import json

import pytest

FOLLOWER_CALIBRATION = {
    "1": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 618, "range_max": 3356},
    "2": {"id": 2, "drive_mode": 0, "homing_offset": 0, "range_min": 851, "range_max": 3112},
    "3": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 1105, "range_max": 3238},
    "4": {"id": 4, "drive_mode": 0, "homing_offset": 0, "range_min": 787, "range_max": 2787},
    "5": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
    "6": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 2039, "range_max": 3421},
}
FOLLOWER_FILE = "so101_follower_calibrate.json"


class PinnedCalibration:
    """The HOME a test runs under; ``use`` swaps in another calibration set."""

    def __init__(self, tmp_path, monkeypatch):
        self._tmp_path = tmp_path
        self._monkeypatch = monkeypatch
        self._homes = 0

    @staticmethod
    def follower(**overrides):
        """A copy of the pinned follower calibration with per-joint field overrides."""
        calibration = copy.deepcopy(FOLLOWER_CALIBRATION)
        for joint, fields in overrides.items():
            calibration[joint].update(fields)
        return calibration

    def use(self, files):
        """Point HOME at a fresh directory whose ``.calibrate`` holds exactly ``files`` (name -> JSON value)."""
        self._homes += 1
        calibrate = self._tmp_path / f"calibration_home_{self._homes}" / ".calibrate"
        calibrate.mkdir(parents=True)
        for name, content in files.items():
            (calibrate / name).write_text(json.dumps(content), encoding="utf-8")
        self._monkeypatch.setenv("HOME", str(calibrate.parent))


@pytest.fixture
def pinned_calibration(tmp_path, monkeypatch):
    """HOME holds only the pinned follower calibration; returns a ``PinnedCalibration``."""
    pinned = PinnedCalibration(tmp_path, monkeypatch)
    pinned.use({FOLLOWER_FILE: FOLLOWER_CALIBRATION})
    return pinned
