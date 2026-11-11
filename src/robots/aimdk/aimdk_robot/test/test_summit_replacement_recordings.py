"""The four replacement recordings load unchanged and agree with their delivery copies."""

import hashlib
from pathlib import Path

import pytest
import yaml
from aimdk_robot.trajectory import load_trajectory

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[3]
PROFILE = PACKAGE / "profiles" / "x2_ultra_omnihand_summit.yaml"


@pytest.mark.parametrize(
    ("name", "frames", "duration", "digest"),
    [
        ("short_wave", 168, 3.34, "ee2393b96020fa6383885fce0857153a6910bd909b945b8a0721d06ed4ea6487"),
        ("invite", 193, 3.84, "846925186ffad04e2c7af85b3e2c0e8195c4c48bc0cad2f0fbe717bc7421c6eb"),
        ("self_intro", 193, 3.84, "f0445e577a560fae14af2c37e3db363b258efe9d8b9257aea03df4a14b2cbd0f"),
        ("explain", 185, 3.68, "bfa952e3f08364d1b4f946f04dc3f0d06e95b9871d4f74f5fa3e37c786ff5b8e"),
    ],
)
def test_replacement_recordings_are_loaded_without_conversion(name, frames, duration, digest):
    config = yaml.safe_load(PROFILE.read_text())["vendor"]["named_motion"]
    path = PROFILE.parent / config["trajectories"][name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    delivery = WORKSPACE / "commissioning" / "motions" / "summit" / name
    assert path.read_bytes() == (delivery / "motion.json").read_bytes()
    assert (path.parent / "skill.json").read_bytes() == (delivery / "skill.json").read_bytes()
    trajectory = load_trajectory(path, max_speed_radps=config["trajectory_max_speed_radps"])
    assert trajectory.target == "both"
    assert len(trajectory.times) == frames
    assert trajectory.duration == pytest.approx(duration)
    assert trajectory.sample(0) == pytest.approx(trajectory.sample(duration))
    assert trajectory.sample_hands(0) == pytest.approx(trajectory.sample_hands(duration))
    assert len(trajectory.sample_hands(duration / 2)) == 20
    assert config["trajectory_blend_speed_radps"] == 0.5
    assert config["trajectory_blend_min_s"] >= 1.5
