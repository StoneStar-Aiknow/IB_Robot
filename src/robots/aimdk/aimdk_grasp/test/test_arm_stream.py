"""The X2 arm's gravity model and its streaming executor settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aimdk_grasp.arm_gravity import MAX_SAG_RAD, ArmGravity, GravityShaper, require_executable
from aimdk_grasp.arm_stream import HOME_COMMAND, x2_arm_executor, x2_stream_config
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.planning_model import BODY_URDF, description_share
from aimdk_grasp.robot_model import arm_limits

from manipulation_execution.grasp_pipeline.contracts import ArmSegment, MotionState, SegmentKind
from manipulation_execution.grasp_pipeline.streaming import CommandSample, JointSample, StreamExecutor

START = np.array([-0.3, 0.3, 0.0, -1.2, 0.0, 0.0, 0.0])
APPROACH = START + np.array([-0.6, 0.1, 0.1, 0.6, 0.1, 0.05, 0.1])
GRASP = APPROACH + np.array([-0.05, 0.0, 0.02, 0.08, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def gravity():
    return ArmGravity.load()


def test_the_gravity_chain_is_the_official_description(gravity):
    """Only kp, the torso tilt and the wrist payload are fitted; the chain must not drift from the URDF."""
    import json

    from aimdk_grasp.robot_model import config_directory

    model = json.loads((config_directory() / "arm_gravity.json").read_text(encoding="utf-8"))
    robot = ET.parse(description_share() / BODY_URDF).getroot()
    for link in model["chain"]:
        joint = robot.find(f"joint[@name='{link['joint']}']")
        inertial = robot.find(f"link[@name='{joint.find('child').get('link')}']/inertial")
        np.testing.assert_allclose(np.fromstring(joint.find("origin").get("xyz"), sep=" "), link["xyz"], atol=1e-12)
        np.testing.assert_allclose(np.fromstring(joint.find("origin").get("rpy"), sep=" "), link["rpy"], atol=1e-12)
        np.testing.assert_allclose(np.fromstring(joint.find("axis").get("xyz"), sep=" "), link["axis"], atol=1e-12)
        assert float(inertial.find("mass").get("value")) == pytest.approx(link["mass"])
        np.testing.assert_allclose(np.fromstring(inertial.find("origin").get("xyz"), sep=" "), link["centre"])
    assert gravity.kp == 40.0


def test_batched_sag_matches_one_waypoint_at_a_time(gravity):
    positions = np.random.default_rng(0).uniform(-1.5, 1.5, (50, 7))
    np.testing.assert_allclose(gravity.sags(positions), [gravity.sag(q) for q in positions], atol=1e-12)


def test_plans_outside_mc_limits_or_with_too_much_sag_are_refused(gravity):
    limits = arm_limits()
    segments = {"approach": np.linspace(START, APPROACH, 20), "grasp": np.linspace(APPROACH, GRASP, 20)}
    worst = require_executable(segments, limits, gravity)
    assert 0 < worst < MAX_SAG_RAD
    beyond = APPROACH.copy()
    beyond[6] = limits[LEFT_ARM[6]][1] + 0.01
    with pytest.raises(ValueError, match="MC joint limits at \\['left_wrist_roll_joint'\\]"):
        require_executable({"approach": np.linspace(START, beyond, 20)}, limits, gravity)
    with pytest.raises(ValueError, match="Predicted sag"):
        require_executable(segments, limits, gravity, max_sag_rad=0.01)


def test_the_shaper_gives_the_predicted_sag_of_the_left_arm_only(gravity):
    positions = np.linspace(START, APPROACH, 5)
    np.testing.assert_allclose(GravityShaper(gravity).raise_for(LEFT_ARM, positions), gravity.sags(positions))
    with pytest.raises(ValueError, match="seven left arm joints"):
        GravityShaper(gravity).raise_for(RIGHT_ARM, positions)


def test_the_x2_settings_come_from_the_robot_configuration():
    config = x2_stream_config()
    assert config.limits == {name: arm_limits()[name] for name in LEFT_ARM}
    assert config.locked_joints == RIGHT_ARM + HEAD
    assert (config.expected_stiffness, config.expected_damping) == (40.0, 2.0)
    assert (config.tracking_error_rad, config.arrival_error_rad) == (0.3, 0.2)
    assert config.home_command == HOME_COMMAND and set(HOME_COMMAND) == set(LEFT_ARM)
    for name, value in HOME_COMMAND.items():
        lower, upper = config.limits[name]
        assert lower + config.limit_margin_rad <= value <= upper - config.limit_margin_rad


class SaggingArm:
    """MC at kp 40: the left arm settles where command - sag(pose) = pose, as the gravity model predicts."""

    channel_joints = LEFT_ARM + RIGHT_ARM

    def __init__(self, gravity, start):
        self.gravity, self.t = gravity, 50.0
        self.measured = np.concatenate([start, np.zeros(7)])
        self.target = self.measured.copy()
        self.target[:7] = self._command_for(start)  # MC's board already holds the static sag

    def _command_for(self, pose):
        return pose + self.gravity.sag(pose)

    def _settle(self, command):
        pose = command.copy()
        for _ in range(30):  # fixed point of pose = command - sag(pose)
            pose = command - self.gravity.sag(pose)
        return pose

    def monotonic(self):
        return self.t

    def wait(self, seconds):
        self.t += seconds
        goal = np.concatenate([self._settle(self.target[:7]), self.target[7:]])
        self.measured += (goal - self.measured) * min(1.0, seconds / 0.05)

    def feedback(self):
        positions = dict(zip(self.channel_joints, self.measured, strict=True))
        positions.update(dict.fromkeys(HEAD, 0.0))
        return JointSample(self.t, positions)

    def command(self):
        gains = dict.fromkeys(self.channel_joints, 40.0), dict.fromkeys(self.channel_joints, 2.0)
        return CommandSample(self.t, dict(zip(self.channel_joints, self.target, strict=True)), *gains)

    def publish(self, positions):
        self.target = np.array(positions, dtype=float)

    def release(self):
        pass

    def refusal(self):
        return None


@pytest.mark.parametrize("shaped", [True, False])
def test_the_gravity_raise_lands_the_measured_arm_on_the_plan(gravity, shaped):
    arm = SaggingArm(gravity, START)
    executor = x2_arm_executor(arm, gravity=gravity)
    if not shaped:
        executor = StreamExecutor(arm, executor.config)
    segment = ArmSegment(
        "approach", SegmentKind.TRAJECTORY, LEFT_ARM, np.linspace(START, APPROACH, 40), np.linspace(0.0, 2.5, 40)
    )
    outcome = executor.execute(segment, carrying=False, halted=lambda: False, cancelled=lambda: False)
    assert outcome.state is MotionState.SUCCEEDED, outcome.reason
    error = float(np.max(np.abs(arm.measured[:7] - APPROACH)))
    # The approach changes the predicted sag by several centimetres of joint travel.
    assert error < 0.01 if shaped else error > 0.04


def test_an_arm_stream_without_both_arms_is_refused(gravity):
    class Partial:
        channel_joints = LEFT_ARM

    with pytest.raises(ValueError, match="does not carry"):
        x2_arm_executor(Partial(), gravity=gravity)
