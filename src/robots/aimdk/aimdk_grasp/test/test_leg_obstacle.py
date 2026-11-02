import json
import shutil

import pytest
from aimdk_grasp.leg_obstacle import FILENAME, build_leg_obstacle, leg_obstacle
from aimdk_grasp.planning_model import description_share
from aimdk_grasp.robot_model import config_directory


def test_the_shipped_obstacle_is_what_the_description_generates():
    pytest.importorskip("trimesh")
    shipped = json.loads((config_directory() / FILENAME).read_text(encoding="utf-8"))
    # Round-trip through JSON: the file stores tuples as lists.
    assert json.loads(json.dumps(build_leg_obstacle(description_share()))) == shipped


def test_an_obstacle_whose_meshes_changed_is_refused(tmp_path):
    shutil.copy(config_directory() / FILENAME, tmp_path / FILENAME)
    data = json.loads((tmp_path / FILENAME).read_text(encoding="utf-8"))
    assert leg_obstacle(tmp_path)["faces"].size == len(data["faces"])
    data["source"]["meshes"]["left_knee_link"]["sha256"] = "0" * 64
    other = tmp_path / "changed"
    other.mkdir()
    (other / FILENAME).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="left_knee_link"):
        leg_obstacle(other)
    # A description that moves a waist, hip or knee joint changes no mesh, only the URDF.
    data = json.loads((tmp_path / FILENAME).read_text(encoding="utf-8"))
    data["source"]["urdf_sha256"] = "0" * 64
    moved = tmp_path / "moved_joint"
    moved.mkdir()
    (moved / FILENAME).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="built from another"):
        leg_obstacle(moved)


def test_the_obstacle_is_read_only_and_in_the_torso_frame():
    obstacle = leg_obstacle()
    assert obstacle["pose"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert not obstacle["vertices"].flags.writeable and not obstacle["faces"].flags.writeable
    assert obstacle["vertices"].shape[1] == 3 and obstacle["faces"].size % 3 == 0
