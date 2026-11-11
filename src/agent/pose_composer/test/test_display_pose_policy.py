"""Display arrival can only opt in compiled non-anchor poses."""

import pytest

from pose_composer.fixed_library import display_pose_policy


@pytest.fixture(scope="session", autouse=True)
def configured():
    return None


@pytest.mark.parametrize("name", ["missing", "arms_down"])
def test_missing_pose_and_functional_home_cannot_opt_in(name):
    library = {
        "poses": {
            "arms_down": {"description": {"left": {"task_pose": "arms_rest"}, "right": {"task_pose": "arms_rest"}}}
        }
    }
    with pytest.raises(ValueError):
        display_pose_policy({"pose_ids": [name], "mode": "playback_only"}, library)


@pytest.mark.parametrize("value", [0, "strict", "ignore_errors", True])
def test_display_policy_requires_explicit_mode(value):
    with pytest.raises(ValueError):
        display_pose_policy({"pose_ids": ["arms_up"], "mode": value})


@pytest.mark.parametrize("name", ["salute", "thinking", "right_hand_on_head"])
def test_near_head_poses_cannot_disable_arrival(name):
    with pytest.raises(ValueError):
        display_pose_policy({"pose_ids": [name], "mode": "playback_only"})
