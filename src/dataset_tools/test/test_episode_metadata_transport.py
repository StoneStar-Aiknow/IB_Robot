"""Episode metadata keeps the execution transport identity (runtime-simulation-transport D7).

A Gazebo-transport runtime description forwarded to the episode recorder (the
``interface_description_json`` hand-off in ``launch_builders/recording.py``)
must survive the dataset metadata round-trip with its ``execution`` and
``transport`` fields intact, so offline consumers can tell physics simulation
apart from the SDK virtual transport.
"""

from __future__ import annotations

import yaml

from robot_config.utils import build_public_lerobot_conversion_metadata
from robot_runtime.interface_description import build_description, description_digest
from robot_runtime.model_metadata import validate_public_conversion_metadata

URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="gripper"/>
  <joint name="1" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-1.0" upper="1.0"/>
  </joint>
  <joint name="6" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="0.0" upper="1.0"/>
  </joint>
</robot>
"""


def _profile():
    return {
        "runtime": {"name": "so101_robot", "version": "0.1.0", "type": "so101"},
        "capabilities": {"joint.state": {"joint_count": 2, "rate_hz": 50.0}},
        "joints": ["1", "6"],
        "arm_joints": ["1"],
        "gripper_joints": ["6"],
        "motion": {"base_link": "base", "ee_link": "gripper", "shoulder_link": "shoulder"},
        "home_positions": {},
        "hardware": {},
    }


def test_gazebo_transport_description_survives_episode_metadata_roundtrip(tmp_path):
    description = build_description(_profile(), transport="gazebo", robot_description=URDF)
    assert description["execution"] == "simulated"
    assert description["transport"] == "gazebo"

    # The recorder snapshots the bound description into the portable conversion
    # metadata it persists for every episode dataset.
    meta = build_public_lerobot_conversion_metadata(
        description["model"], description, {"observation.state": ["1", "6"]}, "degrees"
    )
    fingerprint = meta["conversion_fingerprint"]

    # dataset.yaml persistence shape (episode_recorder._write_dataset_metadata).
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump({"lerobot": {"conversions": {fingerprint: meta}}}, sort_keys=False), encoding="utf-8"
    )

    reloaded = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["lerobot"]["conversions"][fingerprint]
    validate_public_conversion_metadata(reloaded)
    roundtrip = reloaded["description"]
    assert roundtrip["execution"] == "simulated"
    assert roundtrip["transport"] == "gazebo"
    assert description_digest(roundtrip) == roundtrip["digest"]
