"""Real X2 model/planning with fake execution and clock; no ROS graph or hardware."""

import copy

import pytest
from aimdk_robot.pose_backend import model

from pose_composer.composer import ComposeError
from pose_composer.task_poses import TaskPoseRunner
from robot_runtime.task_poses import normalize_task_poses


class MockArm:
    def __init__(self, *, away=False):
        self.now = 10.0
        self.q = dict(model.load().rest)
        if away:
            self.q["left_shoulder_pitch_joint"] -= 0.15
        self.moves = []
        self.cancelled = False
        self.stale = False
        self.no_follow = False
        self.move_failure = None

    def read(self):
        return dict(self.q), {n: self.now - (2 if self.stale else 0) for n in self.q}

    def move(self, plan, deadline):
        self.moves.append(plan)
        assert deadline > self.now
        if self.move_failure:
            return False, self.move_failure, "mock motion failed"
        if not self.no_follow:
            self.q = dict(zip(model.ARM_JOINTS, plan.traj.positions[-1], strict=True))
        return True, "", "mock execution complete"

    def sleep(self, seconds):
        self.now += seconds


def snapshot():
    return normalize_task_poses(
        {
            "schema_version": 1,
            "robot_type": "x2_ultra_omnihand",
            "joint_names": list(model.ARM_JOINTS),
            "poses": {
                "grasp_ready": {
                    "joint_positions": dict(model.load().rest),
                    "stable_sec": 0.05,
                    "timeout_sec": 5.0,
                    "hardware_approved": False,
                }
            },
            "skill_entry_poses": {"pick_object": "grasp_ready"},
        }
    )


def run(arm, config=None, *, simulation=True, name="grasp_ready", digest=None):
    config = config or snapshot()
    runner = TaskPoseRunner(config, simulation=simulation, clock=lambda: arm.now, pause=arm.sleep)
    return runner.execute(
        name, digest or config["digest"], read_state=arm.read, move=arm.move, cancelled=lambda: arm.cancelled
    )


def test_already_at_target_checks_once_without_stability_wait():
    arm = MockArm()
    assert run(arm) is True
    assert arm.moves == []
    assert arm.now == 10.0


def test_measured_arrival_after_planned_motion():
    arm = MockArm(away=True)
    assert run(arm) is False
    assert len(arm.moves) == 1
    assert arm.q == pytest.approx(model.load().rest)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("stale", "TASK_POSE_STALE_STATE"),
        ("cancelled", "POSE_CANCELLED"),
        ("no_follow", "TASK_POSE_NOT_REACHED"),
        ("move_failure", "POSE_RUNTIME_REJECTED"),
    ],
)
def test_preparation_fails_closed(fault, code):
    arm = MockArm(away=True)
    setattr(arm, fault, "POSE_RUNTIME_REJECTED" if fault == "move_failure" else True)
    with pytest.raises(ComposeError) as error:
        run(arm)
    assert error.value.code == code
    assert len(arm.moves) == (0 if fault in ("stale", "cancelled") else 1)


@pytest.mark.parametrize("kwargs", [{"name": "missing"}, {"digest": "old"}, {"simulation": False}])
def test_unknown_changed_or_unapproved_target_cannot_move(kwargs):
    arm = MockArm(away=True)
    with pytest.raises(ComposeError):
        run(arm, **kwargs)
    assert not arm.moves


def test_snapshot_is_independent_of_mutating_the_loaded_file_data():
    config = snapshot()
    old = copy.deepcopy(config)
    arm = MockArm()
    runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    config["poses"]["grasp_ready"]["joint_positions"]["left_elbow_joint"] = -2.0
    assert runner.execute("grasp_ready", old["digest"], read_state=arm.read, move=arm.move, cancelled=lambda: False)
    assert not arm.moves


def test_single_sample_does_not_claim_a_future_stability_window():
    arm = MockArm()

    def read():
        q, stamps = arm.read()
        if 10.02 <= arm.now < 10.08:
            q["left_shoulder_pitch_joint"] += 0.05
        return q, stamps

    config = snapshot()
    runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    assert runner.execute("grasp_ready", config["digest"], read_state=read, move=arm.move, cancelled=lambda: False)
    assert arm.now == 10.0 and not arm.moves
    assert runner.report["arrival_check"]["stable_window_verified"] is False


def test_one_cached_sample_cannot_prove_stable_arrival():
    arm = MockArm()
    config = snapshot()
    runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    cached = arm.read()
    arm.now += 1.0  # Even single-sample confirmation must reject stale feedback.
    with pytest.raises(ComposeError) as error:
        runner.execute(
            "grasp_ready", config["digest"], read_state=lambda: cached, move=arm.move, cancelled=lambda: False
        )
    assert error.value.code == "TASK_POSE_STALE_STATE"
    assert not arm.moves


def test_named_only_ros_node_starts_without_language_model():
    import json
    from pathlib import Path

    import rclpy
    from rclpy.parameter import Parameter

    from pose_composer.node import PoseComposerNode

    root = Path(__file__).resolve().parents[4]
    mjcf = (
        Path.home() / ".cache/ibrobot/agibot_x2_urdf/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.3.0/scene.xml"
    )
    profile = root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"
    rclpy.init()
    node = None
    try:
        node = PoseComposerNode(
            parameter_overrides=[
                Parameter("mjcf_path", value=str(mjcf)),
                Parameter("runtime_profile", value=str(profile)),
                Parameter("task_poses_json", value=json.dumps(snapshot())),
                Parameter("task_pose_simulation", value=True),
                Parameter("trajectory_modes_json", value='["policy_stream"]'),
            ]
        )
        assert node.get_parameter("llm_model").value == ""
        assert node._task_pose_runner.simulation
        assert node._trajectory_modes == {"policy_stream"}
        changed = node.set_parameters([Parameter("task_poses_json", value="{}")])
        assert not changed[0].successful
        assert node._task_pose_runner.snapshot["poses"]
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_out_of_limit_configured_target_is_rejected_without_projection():
    config = snapshot()
    config.pop("digest")
    config["poses"]["grasp_ready"]["joint_positions"]["left_elbow_joint"] = 0.5
    with pytest.raises(ValueError, match="target safety"):
        TaskPoseRunner(config, simulation=True)
