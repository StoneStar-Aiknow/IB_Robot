"""The shipped seed poses all solve inside the robot's travel and are recalled by their phrases."""

from pathlib import Path

import pytest

from pose_composer import safety, spec
from pose_composer.memory import PoseMemory
from pose_composer.model import load
from pose_composer.seeds import load_seeds, read_seed_file

SEEDS = Path(__file__).resolve().parents[1] / "seeds" / "x2_poses.yaml"


def test_every_seed_solves_safely_from_rest():
    _, entries = read_seed_file(SEEDS)
    for entry in entries:
        solved = spec.resolve(entry["description"])
        assert solved is not None, entry["id"]
        q = dict(load().rest, **solved[1])
        assert safety.check_pose(q).ok, entry["id"]


def test_seeds_load_once_per_version_and_answer_their_phrases(tmp_path):
    memory = PoseMemory(tmp_path / "m.sqlite3", "fp")
    try:
        assert load_seeds(memory, str(SEEDS)) > 0
        assert load_seeds(memory, str(SEEDS)) == 0
        memory.revalidate(spec.resolve)
        for phrase in ("双手向前伸直", "两只手向前伸直", "敬个礼", "投降"):
            record = memory.lookup(phrase, confirmed_only=True)
            assert record is not None and record.state == "pinned", phrase
    finally:
        memory.close()


@pytest.mark.parametrize("name,passive", [("right_forward", "left"), ("left_forward", "right")])
def test_single_arm_seed_returns_the_other_arm_to_configured_home(name, passive):
    _, entries = read_seed_file(SEEDS)
    seed = next(s for s in entries if s["id"] == name)
    robot = load()
    home = robot.task_poses["poses"]["grasp_ready"]["joint_positions"]
    start = dict(home)
    start[f"{passive}_elbow_joint"] = -1.7
    target = spec.compose(seed["description"], start).frames[0]
    joints = [n for n in home if n.startswith(passive + "_")]
    assert all(target[n] == home[n] for n in joints)
    assert target[f"{passive}_elbow_joint"] != start[f"{passive}_elbow_joint"]


def test_arms_down_is_exact_home_and_not_profile_rest():
    _, entries = read_seed_file(SEEDS)
    seed = next(s for s in entries if s["id"] == "arms_down")
    robot = load()
    target = spec.compose(seed["description"], dict(robot.rest)).frames[0]
    assert target == robot.task_poses["poses"]["grasp_ready"]["joint_positions"]


def test_named_home_fails_if_unconfigured_and_never_falls_back_to_current(monkeypatch):
    from dataclasses import replace

    from aimdk_robot.pose_backend import model

    robot = load()
    monkeypatch.setattr(model, "_ROBOT", replace(robot, task_poses={}))
    with pytest.raises(spec.SpecError, match="not configured"):
        spec.compose({"kind": "pose", "left": {"task_pose": "grasp_ready"}}, dict(robot.rest))


def test_reference_cannot_be_combined_with_keep_or_ik():
    with pytest.raises(spec.SpecError, match="exclusive"):
        spec.compose({"kind": "pose", "left": {"task_pose": "grasp_ready", "keep": True}}, dict(load().rest))


def test_home_changes_invalidate_seed_memory_fingerprint(monkeypatch):
    import copy
    from dataclasses import replace

    from aimdk_robot.pose_backend import model

    from pose_composer.seeds import fingerprint
    from robot_runtime.task_poses import normalize_task_poses

    robot = load()
    path = (
        Path.home() / ".cache/ibrobot/agibot_x2_urdf/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.3.0/scene.xml"
    )
    before = fingerprint("prompt", "model", str(path))
    changed = copy.deepcopy(robot.task_poses)
    changed.pop("digest")
    changed["poses"]["grasp_ready"]["joint_positions"]["left_shoulder_pitch_joint"] += 0.02
    monkeypatch.setattr(model, "_ROBOT", replace(robot, task_poses=normalize_task_poses(changed)))
    assert fingerprint("prompt", "model", str(path)) != before
