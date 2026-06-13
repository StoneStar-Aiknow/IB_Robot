import importlib.util
import sys
from pathlib import Path

import numpy as np
from sensor_msgs.msg import Image, JointState

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "embodied_motion_effect_test.py"
SPEC = importlib.util.spec_from_file_location("embodied_motion_effect_test", SCRIPT_PATH)
motion_effect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_effect
SPEC.loader.exec_module(motion_effect)


def _joint_state(names, positions):
    msg = JointState()
    msg.name = list(names)
    msg.position = list(positions)
    return msg


def _mono_image(values):
    array = np.array(values, dtype=np.uint8)
    msg = Image()
    msg.height, msg.width = array.shape
    msg.encoding = "mono8"
    msg.step = msg.width
    msg.data = array.tobytes()
    return msg


def test_joint_position_delta_uses_matching_joint_names():
    before = motion_effect.joint_position_map(_joint_state(["1", "2", "3"], [0.0, 0.2, -0.1]))
    after = motion_effect.joint_position_map(_joint_state(["3", "1", "2"], [-0.1, 0.04, 0.2]))

    max_delta, moved_names = motion_effect.max_joint_delta(before, after)

    assert max_delta == 0.04
    assert moved_names == ["1"]


def test_max_joint_delta_across_samples_catches_return_to_start_motion():
    baseline = motion_effect.joint_position_map(_joint_state(["1", "2"], [0.0, 0.0]))
    samples = [
        motion_effect.joint_position_map(_joint_state(["1", "2"], [0.0, 0.0])),
        motion_effect.joint_position_map(_joint_state(["1", "2"], [0.0, 0.25])),
        motion_effect.joint_position_map(_joint_state(["1", "2"], [0.0, 0.0])),
    ]

    max_delta, moved_names = motion_effect.max_joint_delta_across_samples(baseline, samples)

    assert max_delta == 0.25
    assert moved_names == ["2"]


def test_mean_image_delta_reports_pixel_change():
    before = _mono_image([[0, 0], [0, 0]])
    after = _mono_image([[0, 10], [20, 30]])

    assert motion_effect.mean_image_delta(before, after) == 15.0


def test_motion_effect_requires_joint_delta_and_optional_image_delta():
    metrics = motion_effect.MotionEffectMetrics(
        max_joint_delta_rad=0.02,
        mean_image_delta=0.0,
        image_samples_before=1,
        image_samples_after=2,
        joint_names_moved=["1"],
    )

    assert motion_effect.motion_effect_passed(
        metrics,
        min_joint_delta_rad=0.005,
        min_image_delta=2.0,
        require_image_change=False,
    )
    assert not motion_effect.motion_effect_passed(
        metrics,
        min_joint_delta_rad=0.005,
        min_image_delta=2.0,
        require_image_change=True,
    )
