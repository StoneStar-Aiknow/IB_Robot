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

"""The public robot model the LeKiwi runtime hands to its facade.

The runtime launch renders the whole robot once and builds the interface
description from it, as the SO-101 runtime does. Without that the facade has no
``model`` and consumers cannot read joint groups, limits or frames from the
runtime.
"""

from pathlib import Path

import pytest

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import render_robot_description, resolve_profile_argument
from robot_runtime.profile import load_profile


def _description(name: str, simulated: bool) -> dict:
    path = resolve_profile_argument("lekiwi_robot", name)
    profile = load_profile(path)
    robot_description = render_robot_description(profile, path, simulated)
    # build_description takes use_sim (upstream runtime-simulation-platform:
    # one selection, one name; the simulated kwarg is gone).
    return build_description(profile, use_sim=simulated, robot_description=robot_description)


@pytest.mark.parametrize("simulated", (True, False), ids=("simulated", "physical"))
def test_nav_grasp_publishes_arm_gripper_and_wheels(simulated):
    groups = _description("lekiwi_nav_grasp", simulated)["model"]["joint_groups"]
    assert groups["base"] == ["7", "8", "9"]
    assert len(groups["arm"]) == 5 and len(groups["gripper"]) == 1
    assert groups["all"] == [*groups["arm"], *groups["gripper"], *groups["base"]]


@pytest.mark.parametrize("simulated", (True, False), ids=("simulated", "physical"))
def test_base_only_profile_publishes_wheels_only(simulated):
    groups = _description("lekiwi_lidar", simulated)["model"]["joint_groups"]
    assert groups["base"] == ["7", "8", "9"]
    assert groups["arm"] == [] and groups["gripper"] == []


def test_nav_grasp_home_positions_are_the_actuator_reset_positions():
    # The public semantic HOME must equal the actuator startup pose for the
    # arm joints: home_positions projects hardware.reset_positions for joints
    # 1-5 (mirroring so101_single_arm), so the embodied arm surface and the
    # pick executor's joint-5 HOME gating read one source. The projection is
    # additive — actuator startup still drives hardware.reset_positions,
    # which this addition does not touch.
    profile = load_profile(resolve_profile_argument("lekiwi_robot", "lekiwi_nav_grasp"))
    reset = profile["hardware"]["reset_positions"]
    home = profile["home_positions"]
    assert sorted(home) == sorted(profile["arm_joints"])
    assert home == {name: reset[name] for name in profile["arm_joints"]}
    assert home == {"1": 0.0, "2": -1.5854, "3": 1.5708, "4": 0.7854, "5": 0.0}


def test_nav_grasp_declares_the_public_ee_pose_stream():
    # The embodied arm surface requires a public end-effector pose stream;
    # so101_motion's motion server (part of this runtime) publishes it. The
    # declaration mirrors so101_single_arm (openspec change
    # cleanup-manipulation-so101-providers D5).
    profile = load_profile(resolve_profile_argument("lekiwi_robot", "lekiwi_nav_grasp"))
    assert "motion.fk" in profile["capabilities"]
    assert profile["interfaces"]["motion.ee_pose"] == {
        "capability": "motion.fk",
        "kind": "topic",
        "direction": "publish",
        "endpoint": "/robot_status/ee_pose",
        "message_type": "geometry_msgs/msg/PoseStamped",
        "frame_id": "base",
        "qos": {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10},
    }


@pytest.mark.parametrize("simulated", (True, False), ids=("simulated", "physical"))
def test_nav_grasp_description_exposes_home_positions_and_ee_pose(simulated):
    # Both D5 profile additions are part of the public description (and
    # therefore its digest): the model's semantic home positions and the
    # declared motion.ee_pose stream.
    profile = load_profile(resolve_profile_argument("lekiwi_robot", "lekiwi_nav_grasp"))
    description = _description("lekiwi_nav_grasp", simulated)
    assert description["model"]["home_positions"] == {
        "1": 0.0,
        "2": -1.5854,
        "3": 1.5708,
        "4": 0.7854,
        "5": 0.0,
    }
    assert description["interfaces"]["motion.ee_pose"] == profile["interfaces"]["motion.ee_pose"]


def test_runtime_launch_passes_the_model_to_the_facade():
    # Source-level pin: the description must reach runtime_stack_actions,
    # otherwise the facade rebuilds one without a model.
    launch = (Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py").read_text()
    assert "robot_description=robot_description" in launch
    assert "interface_description=description" in launch
